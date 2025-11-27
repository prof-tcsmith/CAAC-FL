"""
Shared model architectures for all experiment levels.

Supports:
- MNIST/Fashion-MNIST (28x28, 1 channel)
- CIFAR-10 (32x32, 3 channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.

    Supports different input sizes and channels:
    - CIFAR-10: 32x32x3 -> fc_size = 64*8*8 = 4096
    - MNIST/Fashion-MNIST: 28x28x1 -> fc_size = 64*7*7 = 3136

    Architecture:
    - Conv1: in_channels -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - FC1: fc_size -> 128
    - FC2: 128 -> num_classes
    """

    def __init__(self, num_classes=10, in_channels=3, input_size=32):
        super(SimpleCNN, self).__init__()

        self.in_channels = in_channels
        self.input_size = input_size

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate FC input size based on input dimensions
        # After 2 pooling layers: size / 4
        fc_size = 64 * (input_size // 4) * (input_size // 4)
        self.fc_size = fc_size

        # Fully connected layers
        self.fc1 = nn.Linear(fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # size -> size/2

        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # size/2 -> size/4

        # Flatten
        x = x.view(-1, self.fc_size)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_parameters(self):
        """Return model parameters as a list of tensors."""
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of tensors."""
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()

    def get_gradients(self):
        """Return model gradients as a list of tensors."""
        return [param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for param in self.parameters()]


class MLP(nn.Module):
    """
    Multi-layer Perceptron for simpler experiments.

    Architecture:
    - FC1: 3072 -> 512
    - FC2: 512 -> 256
    - FC3: 256 -> 10
    """

    def __init__(self, input_dim=3072, hidden_dims=[512, 256], num_classes=10):
        super(MLP, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.model(x)

    def get_parameters(self):
        """Return model parameters as a list of tensors."""
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of tensors."""
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()

    def get_gradients(self):
        """Return model gradients as a list of tensors."""
        return [param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for param in self.parameters()]


def create_model(model_name='cnn', **kwargs):
    """
    Factory function to create models.

    Args:
        model_name: 'cnn' or 'mlp'
        **kwargs: Additional arguments passed to model constructor

    Returns:
        PyTorch model
    """
    if model_name.lower() == 'cnn':
        return SimpleCNN(**kwargs)
    elif model_name.lower() == 'mlp':
        return MLP(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def create_model_for_dataset(dataset_name: str, num_classes: int = 10):
    """
    Create appropriate model for a given dataset.

    Args:
        dataset_name: One of 'mnist', 'fashion_mnist', 'cifar10'
        num_classes: Number of output classes

    Returns:
        PyTorch model configured for the dataset
    """
    dataset_name = dataset_name.lower().replace('-', '_')

    if dataset_name in ['mnist', 'fashion_mnist']:
        return SimpleCNN(
            num_classes=num_classes,
            in_channels=1,
            input_size=28
        )
    elif dataset_name == 'cifar10':
        return SimpleCNN(
            num_classes=num_classes,
            in_channels=3,
            input_size=32
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation for different datasets
    print("Testing SimpleCNN for CIFAR-10 (32x32x3):")
    cnn_cifar = create_model_for_dataset('cifar10')
    print(f"  Parameters: {count_parameters(cnn_cifar):,}")
    dummy_cifar = torch.randn(4, 3, 32, 32)
    output = cnn_cifar(dummy_cifar)
    print(f"  Input shape: {dummy_cifar.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nTesting SimpleCNN for MNIST (28x28x1):")
    cnn_mnist = create_model_for_dataset('mnist')
    print(f"  Parameters: {count_parameters(cnn_mnist):,}")
    dummy_mnist = torch.randn(4, 1, 28, 28)
    output = cnn_mnist(dummy_mnist)
    print(f"  Input shape: {dummy_mnist.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nTesting SimpleCNN for Fashion-MNIST (28x28x1):")
    cnn_fmnist = create_model_for_dataset('fashion_mnist')
    print(f"  Parameters: {count_parameters(cnn_fmnist):,}")
    output = cnn_fmnist(dummy_mnist)
    print(f"  Output shape: {output.shape}")

    print("\nTesting MLP:")
    mlp = MLP()
    print(f"  Parameters: {count_parameters(mlp):,}")
    output = mlp(dummy_cifar)
    print(f"  Output shape: {output.shape}")
