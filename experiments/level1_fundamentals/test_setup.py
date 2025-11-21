"""
Test script to verify Level 1 setup before running full experiments.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import flwr
        print("  ✓ Flower imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import Flower: {e}")
        return False

    try:
        import torchvision
        print("  ✓ torchvision imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import torchvision: {e}")
        return False

    try:
        from shared.models import SimpleCNN
        print("  ✓ SimpleCNN imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import SimpleCNN: {e}")
        return False

    try:
        from shared.data_utils import load_cifar10, partition_data_iid
        print("  ✓ Data utilities imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import data utilities: {e}")
        return False

    try:
        from shared.metrics import evaluate_model, MetricsLogger
        print("  ✓ Metrics utilities imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import metrics utilities: {e}")
        return False

    return True


def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    try:
        from shared.models import SimpleCNN, count_parameters

        model = SimpleCNN(num_classes=10)
        num_params = count_parameters(model)
        print(f"  ✓ Model created with {num_params:,} parameters")

        # Test forward pass
        dummy_input = torch.randn(4, 3, 32, 32)
        output = model(dummy_input)
        assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
        print(f"  ✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")

        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_data_loading():
    """Test data loading and partitioning."""
    print("\nTesting data loading...")
    try:
        from shared.data_utils import load_cifar10, partition_data_iid

        # Load small subset for testing
        print("  Loading CIFAR-10 (this may take a moment on first run)...")
        train_dataset, test_dataset = load_cifar10()
        print(f"  ✓ Data loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")

        # Test partitioning
        num_clients = 10
        client_dict = partition_data_iid(train_dataset, num_clients, seed=42)
        print(f"  ✓ Data partitioned into {num_clients} clients")

        # Verify partition sizes
        total_samples = sum(len(indices) for indices in client_dict.values())
        assert total_samples == len(train_dataset), "Partition sizes don't sum to dataset size"
        print(f"  ✓ Partition verified: {total_samples} total samples")

        return True
    except Exception as e:
        print(f"  ✗ Data loading test failed: {e}")
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    try:
        from shared.metrics import compute_detection_metrics

        # Test detection metrics
        true_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        predicted_labels = [0, 0, 1, 0, 1, 1, 0, 1]
        metrics = compute_detection_metrics(true_labels, predicted_labels)

        assert 'tpr' in metrics, "Missing TPR metric"
        assert 'fpr' in metrics, "Missing FPR metric"
        print("  ✓ Detection metrics computed successfully")

        return True
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        return False


def test_flower_strategies():
    """Test that Flower strategies are available."""
    print("\nTesting Flower strategies...")
    try:
        from flwr.server.strategy import FedAvg, FedMedian

        print("  ✓ FedAvg strategy available")
        print("  ✓ FedMedian strategy available")

        return True
    except ImportError as e:
        print(f"  ✗ Strategy import failed: {e}")
        print("  Note: If FedMedian is not available, we can implement it manually")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Level 1 Setup Verification")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model", test_model()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Metrics", test_metrics()))
    results.append(("Flower Strategies", test_flower_strategies()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Ready to run experiments.")
        print("  Run: bash run_all.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before running experiments.")
        return 1


if __name__ == "__main__":
    exit(main())
