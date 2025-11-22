"""
Shared metrics and evaluation utilities.
"""

# Suppress PyTorch pin_memory deprecation warnings (from PyTorch internals)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch.utils.data')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple
import json
import csv
import os


def evaluate_model(model, dataloader, device='cpu', criterion=None):
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)

    Returns:
        dict: {
            'accuracy': float,
            'loss': float (if criterion provided),
            'num_samples': int
        }
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            num_batches += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    result = {
        'accuracy': accuracy,
        'num_samples': total,
        'num_correct': correct
    }

    if criterion is not None:
        result['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    return result


def train_model(model, dataloader, optimizer, criterion, device='cpu', epochs=1):
    """
    Train model for specified epochs.

    Args:
        model: PyTorch model
        dataloader: DataLoader for training
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epochs: Number of epochs

    Returns:
        dict: Training statistics
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return {
        'train_loss': total_loss / len(dataloader) / epochs,
        'train_accuracy': accuracy,
        'num_samples': total
    }


def compute_gradient_norm(model):
    """
    Compute L2 norm of model gradients.

    Args:
        model: PyTorch model

    Returns:
        float: Gradient norm
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return np.sqrt(total_norm)


def compute_model_norm(model):
    """
    Compute L2 norm of model parameters.

    Args:
        model: PyTorch model

    Returns:
        float: Parameter norm
    """
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2

    return np.sqrt(total_norm)


def compute_cosine_similarity(grad1, grad2):
    """
    Compute cosine similarity between two gradient vectors.

    Args:
        grad1: First gradient (list of tensors or flattened tensor)
        grad2: Second gradient (list of tensors or flattened tensor)

    Returns:
        float: Cosine similarity
    """
    # Flatten and concatenate if needed
    if isinstance(grad1, list):
        grad1 = torch.cat([g.flatten() for g in grad1])
    if isinstance(grad2, list):
        grad2 = torch.cat([g.flatten() for g in grad2])

    # Compute cosine similarity
    dot_product = torch.dot(grad1, grad2)
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return (dot_product / (norm1 * norm2)).item()


class MetricsLogger:
    """
    Logger for experiment metrics.
    """

    def __init__(self, log_dir, experiment_name):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'test_loss': []
        }

        os.makedirs(log_dir, exist_ok=True)

    def log_round(self, round_num, train_loss=None, train_acc=None,
                  test_acc=None, test_loss=None, **kwargs):
        """
        Log metrics for a round.

        Args:
            round_num: Round number
            train_loss: Training loss
            train_acc: Training accuracy
            test_acc: Test accuracy
            test_loss: Test loss
            **kwargs: Additional metrics to log
        """
        self.metrics['rounds'].append(round_num)

        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        if train_acc is not None:
            self.metrics['train_accuracy'].append(train_acc)
        if test_acc is not None:
            self.metrics['test_accuracy'].append(test_acc)
        if test_loss is not None:
            self.metrics['test_loss'].append(test_loss)

        # Handle additional metrics
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def save_csv(self):
        """Save metrics to CSV file."""
        csv_path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.csv")

        # Get all metric keys
        keys = sorted(self.metrics.keys())

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)

            # Write rows
            num_rows = len(self.metrics['rounds'])
            for i in range(num_rows):
                row = []
                for key in keys:
                    if i < len(self.metrics[key]):
                        row.append(self.metrics[key][i])
                    else:
                        row.append('')
                writer.writerow(row)

        print(f"Metrics saved to {csv_path}")

    def save_json(self):
        """Save metrics to JSON file."""
        json_path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")

        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Metrics saved to {json_path}")

    def get_metrics(self):
        """Return metrics dictionary."""
        return self.metrics


def compute_detection_metrics(true_labels, predicted_labels):
    """
    Compute detection metrics (TPR, FPR, F1, etc.).

    Args:
        true_labels: Ground truth (0=benign, 1=Byzantine)
        predicted_labels: Predictions (0=benign, 1=Byzantine)

    Returns:
        dict: Detection metrics
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()

    # Metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


if __name__ == "__main__":
    print("Testing metrics utilities...")

    # Test detection metrics
    true_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    predicted_labels = [0, 0, 1, 0, 1, 1, 0, 1]

    metrics = compute_detection_metrics(true_labels, predicted_labels)
    print("\nDetection Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test metrics logger
    print("\nTesting MetricsLogger:")
    logger = MetricsLogger('/tmp', 'test_experiment')
    logger.log_round(1, train_loss=1.5, train_acc=60.0, test_acc=55.0)
    logger.log_round(2, train_loss=1.2, train_acc=65.0, test_acc=60.0)
    logger.save_csv()
    logger.save_json()
